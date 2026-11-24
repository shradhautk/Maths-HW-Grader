import LegalLayout from './LegalLayout'

/**
 * Converted from legal/privacy.html. The wording is unchanged; only the page
 * around it (LegalLayout: header, document switcher, "On this page") is new.
 */
export default function Privacy() {
  return (
    <LegalLayout
      title="Privacy Policy"
      effective="Effective Date: 12/05/2025"
      intro={
        <>
          <p>
            GradeAnt ("GradeAnt", "we", "us", or "our"), affiliated with Missouri S&T, is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you use our website www.grade-ant.com and our AI-powered homework grading services (collectively, the "Service").
          </p>
          <p>
            Please read this Privacy Policy carefully. By using the Service, you agree to the terms of this Privacy Policy. If you do not agree, please do not access the Service.
          </p>
        </>
      }
    >
      <div id="section1" className="lg-section">
        <h2>
          1. Information We Collect
        </h2>
        <p>We may collect the following types of information:</p>
        <strong>Personal Information:</strong>
        <ul>
          <li>
            <strong>Account Information:</strong>
            {' '}When you register, we collect your name, email address, password, and your affiliated university.
          </li>
          <li>
            <strong>Homework Content:</strong>
            {' '}We collect the actual homework files (PDFs) you upload for grading ("User Content"). This may contain personal information depending on what you include in your submissions.
          </li>
          <li>
            <strong>Grading Results:</strong>
            {' '}We collect the AI-generated feedback and preliminary grades associated with your submissions.
          </li>
        </ul>
        <strong>Usage Data:</strong>
        <ul>
          <li>
            We might collect information about how you access and use the Service, such as your IP address, browser type, device information, operating system, pages visited, features used, access times, and timestamps.
          </li>
        </ul>
        <strong>Cookies and Tracking Technologies:</strong>
        <ul>
          <li>
            We use cookies (small text files stored on your device) and similar tracking technologies to operate and improve the Service, understand usage, and manage sessions. This may include essential cookies (for functionality) and analytics cookies. You can typically control cookie preferences through your browser settings.
          </li>
        </ul>
      </div>
      <div id="section2" className="lg-section">
        <h2>
          2. How We Use Your Information
        </h2>
        <p>We use the information we collect for the following purposes:</p>
        <ul>
          <li>
            <strong>To Provide and Manage the Service:</strong>
            {' '}To operate the AI grading functionality, create and manage your account, process your submissions, and display results.
          </li>
          <li>
            <strong>To Communicate With You:</strong>
            {' '}To send service-related notifications (e.g., updates, security alerts), respond to support requests, and provide information about the Service. We currently do not send marketing emails.
          </li>
          <li>
            <strong>For Research Purposes:</strong>
            {' '}To use anonymized or aggregated User Content (homework submissions and results) for educational research, aiming to understand learning patterns and improve pedagogical methods, consistent with ethical guidelines (e.g., IRB standards). We take steps like removing direct identifiers (names) from submitted documents used for research. We do not currently use your specific homework submissions to train our primary AI grading models, though research findings may inform future model improvements. You may have the option to opt-out of this research use during account creation.
          </li>
          <li>
            <strong>To Improve the Service:</strong>
            {' '}To analyze usage patterns (using aggregated Usage Data) to understand how the Service is used, identify areas for improvement, and enhance user experience.
          </li>
          <li>
            <strong>For Security and Compliance:</strong>
            {' '}To maintain the security and integrity of the Service, prevent fraud and abuse, and comply with applicable legal obligations.
          </li>
        </ul>
        <div className="lg-callout">
          <p>
            
            <strong>Important:</strong>
            {' '}We do not use your homework submissions to train our primary AI grading models without your consent. You can opt-out of research use during account creation.
          </p>
        </div>
      </div>
      <div id="section3" className="lg-section">
        <h2>
          3. How We Share Your Information
        </h2>
        <p>
          We do not sell your personal information. We may share your information in the following circumstances:
        </p>
        <ul>
          <li>
            <strong>With Educational Personnel:</strong>
            {' '}Your homework submissions and the corresponding AI-generated grades/feedback will be accessible to authorized personnel (e.g., teachers, TAs) associated with your course or institution who use the GradeAnt platform.
          </li>
          <li>
            <strong>Service Providers:</strong>
            {' '}We share information with third-party vendors who perform services on our behalf, such as:
            <ul>
              <li>
                Cloud Hosting Providers (e.g., Amazon Web Services - AWS) for data storage and processing.
              </li>
              <li>
                Analytics Providers (e.g., Google Analytics - using aggregated/anonymized data where feasible) to help us understand Service usage.
              </li>
              <li>Email Service Providers (for service-related communications).</li>
            </ul>
          </li>
          <li>
            <strong>Third-Party AI Service Providers</strong>
            {' '}(e.g., models like Google Gemini, OpenAI GPT): Portions of your submitted content may be processed by these external APIs as part of the grading pipeline. We aim to use providers with strong privacy commitments, but their handling of data is subject to their respective privacy policies.
          </li>
          <li>
            <strong>For Research:</strong>
            {' '}Anonymized or aggregated data may be shared with research partners or published as part of research findings, consistent with IRB guidelines and ethical practices.
          </li>
          <li>
            <strong>Legal Requirements:</strong>
            {' '}We may disclose your information if required by law, subpoena, or other legal process, or if we believe in good faith that disclosure is necessary to protect our rights, protect your safety or the safety of others, investigate fraud, or respond to a government request.
          </li>
          <li>
            <strong>Business Transfers:</strong>
            {' '}In the event of a merger, acquisition, or asset sale involving GradeAnt, your information may be transferred as part of the transaction, subject to the commitments made in this Privacy Policy.
          </li>
        </ul>
      </div>
      <div id="section4" className="lg-section">
        <h2>
          4. Data Security
        </h2>
        <p>
          We implement reasonable administrative, technical, and physical security measures designed to protect your personal information from unauthorized access, use, alteration, or disclosure. These measures include encryption of data in transit and at rest where feasible, access controls, and efforts to anonymize personal information (like names) within submitted answer sheets before use in research datasets.
        </p>
        <div className="lg-callout">
          <p>
            However, no internet transmission or electronic storage is 100% secure, so we cannot guarantee absolute security.
          </p>
        </div>
      </div>
      <div id="section5" className="lg-section">
        <h2>
          5. Data Retention
        </h2>
        <ul>
          <li>
            <strong>Account Data:</strong>
            {' '}We retain your account information for as long as your account is active or as needed to provide you the Service. We will delete inactive account data approximately 6 months after the last recorded use.
          </li>
          <li>
            <strong>Homework Submissions and Results:</strong>
            {' '}We retain submitted homework and grading results for approximately 1 year to allow access by you and your institution and for potential research purposes as described above. After this period, it may be deleted or fully anonymized.
          </li>
        </ul>
        <p>
          We may retain certain information longer if required by law or for legitimate business purposes like resolving disputes or enforcing our agreements.
        </p>
      </div>
      <div id="section6" className="lg-section">
        <h2>
          6. Your Rights and Choices
        </h2>
        <p>
          Depending on your location and applicable law, you may have certain rights regarding your personal information:
        </p>
        <ul>
          <li>
            <strong>Access, Correction, Deletion:</strong>
            {' '}You can access and update your basic account information (name, email, university) through your user profile settings. You may request deletion of your account by contacting us.
          </li>
          <li>
            <strong>Homework Submissions:</strong>
            {' '}Once submitted, homework cannot typically be deleted by the user, but it may potentially be replaced with an updated version before a deadline, subject to course rules. Deletion requests for specific submissions would need to be handled administratively, potentially requiring institutional approval.
          </li>
          <li>
            <strong>Opt-Out of Research Use:</strong>
            {' '}You may have the option during account creation to opt-out of having your anonymized data used for research purposes. If you wish to change this preference later, please contact us.
          </li>
          <li>
            <strong>Marketing Communications:</strong>
            {' '}We currently do not send marketing emails. If we do in the future, you will have the option to opt-out.
          </li>
        </ul>
        <p>
          <strong>Exercising Rights:</strong>
          {' '}To exercise applicable rights, please contact us at{' '}
          <a href="mailto:contact@gradeant.com">contact@gradeant.com</a>. We may need to verify your identity before processing your request.
        </p>
      </div>
      <div id="section7" className="lg-section">
        <h2>
          7. Children's Privacy
        </h2>
        <p>
          The Service is not intended for individuals under the age of 16. We do not knowingly collect personal information from children under 16. If we become aware that we have inadvertently collected personal information from a child under 16 without appropriate consent, we will take steps to delete such information.
        </p>
      </div>
      <div id="section8" className="lg-section">
        <h2>
          8. International Data Transfers
        </h2>
        <p>
          Your information will be processed and stored on servers located in the United States. If you are accessing the Service from outside the U.S., please be aware that your information will be transferred to, stored, and processed in the U.S., where data protection laws may differ from those in your jurisdiction. By using the Service, you consent to this transfer.
        </p>
      </div>
      <div id="section9" className="lg-section">
        <h2>
          9. Changes to this Privacy Policy
        </h2>
        <p>
          We may update this Privacy Policy from time to time. We will notify you of any significant changes by sending an email to the address associated with your account or by posting a prominent notice on the Service prior to the change becoming effective. We encourage you to review this Privacy Policy periodically.
        </p>
      </div>
      <div id="section10" className="lg-section">
        <h2>
          10. Contact Information
        </h2>
        <p>
          If you have any questions or concerns about this Privacy Policy or our data practices, please contact us at:
        </p>
        <div className="lg-contact">
          <p>
            Email:{' '}
            <strong>contact@gradeant.com</strong>
          </p>
        </div>
      </div>
    </LegalLayout>
  )
}
